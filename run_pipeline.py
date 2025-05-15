import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import json
import config      # Import config
import utils       # Import utils
import captioning
import revision    # Import revision (now includes alignment)
import qa_generation
from qa_filter import filter_qa_list # Import the filtering function

def process_single_directory(directory_path, run_flags, qa_metadata=None, file_lock=None):
    """
    Runs the selected processing steps for a single video directory.
    """
    print(f"\n===========================================")
    print(f"Processing Directory: {directory_path}")
    print(f"Run Flags: {run_flags}")
    print(f"===========================================")
    start_time = time.time()
    success_flags = {}

    video_id = os.path.basename(directory_path)

    # --- Step 1: Initial Captioning ---
    if run_flags.get('captioning', False):
        # (Keep existing captioning logic) ...
        with ThreadPoolExecutor(max_workers=2) as executor:
             futures = {}
             if run_flags.get('video_caption', False):
                 futures[executor.submit(captioning.generate_video_captions, directory_path)] = 'video_caption'
             if run_flags.get('audio_caption', False):
                 use_gemini_audio = run_flags.get('use_gemini_audio', True)
                 futures[executor.submit(captioning.generate_audio_captions, directory_path, use_gemini=use_gemini_audio)] = 'audio_caption'

             captioning_futures_completed = 0
             for future in as_completed(futures):
                 step_name = futures[future]
                 try:
                     success_flags[step_name] = future.result()
                     captioning_futures_completed += 1
                 except Exception as e:
                     print(f"Error during {step_name} for {directory_path}: {e}")
                     success_flags[step_name] = False

             # Check overall captioning success only if all submitted futures completed
             if len(futures) > 0 and captioning_futures_completed == len(futures):
                 requested_caption_steps = [s for s in ['video_caption', 'audio_caption'] if run_flags.get(s, False)]
                 success_flags['captioning'] = all(success_flags.get(step, False) for step in requested_caption_steps)
             elif len(futures) == 0: # No captioning steps requested within the block
                  success_flags['captioning'] = True
             else: # Not all futures completed (e.g., due to errors)
                  success_flags['captioning'] = False
    else:
        print("Skipping captioning step.")
        success_flags['captioning'] = True # Mark as success if skipped for dependency chain

    # --- Step 2: Revision ---
    # Run revision only if captioning was successful *or skipped* and revision is requested
    if run_flags.get('revision', False) and success_flags.get('captioning', True): # Depends on captioning *block* success/skip
         with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            # Video revision needs *initial* video captions
            can_run_video_revision = run_flags.get('video_revision', False) and os.path.exists(os.path.join(directory_path, "video_detail_captions.txt"))
            if run_flags.get('video_revision', False):
                if can_run_video_revision:
                     futures[executor.submit(revision.revise_video_consistency, directory_path)] = 'video_revision'
                else:
                     print("Skipping video revision as prerequisite 'video_detail_captions.txt' is missing.")
                     success_flags['video_revision'] = False # Mark specific step as failed/skipped

            # Audio revision needs *initial* audio captions and *consistent* video captions
            # Check for video_consistent_captions *after* video revision future might have run
            # This dependency makes threading tricky - safer to run sequentially or check after video revision finishes
            # For simplicity here, check based on flag + potential file existence
            # A better way: run audio revision *after* video revision future completes successfully

            # --- Simplified Revision Execution (Sequential within block for dependency) ---
            revision_success = True # Assume success unless a step fails

            if run_flags.get('video_revision', False):
                 if can_run_video_revision:
                     try:
                          video_rev_ok = revision.revise_video_consistency(directory_path)
                          success_flags['video_revision'] = video_rev_ok
                          if not video_rev_ok: revision_success = False
                     except Exception as e:
                          print(f"Error during video_revision for {directory_path}: {e}")
                          success_flags['video_revision'] = False
                          revision_success = False
                 else:
                      # Already printed message, just mark flags
                      success_flags['video_revision'] = False
                      revision_success = False # Cannot proceed if video rev fails

            # Run Audio Revision only if video revision was successful (or wasn't requested but audio rev is)
            # and necessary files exist
            if run_flags.get('audio_revision', False) and revision_success:
                 can_run_audio_revision = os.path.exists(os.path.join(directory_path, "audio_detail_captions.txt")) and \
                                      os.path.exists(os.path.join(directory_path, "video_consistent_captions.txt"))
                 if can_run_audio_revision:
                      try:
                           audio_rev_ok = revision.revise_audio_captions(directory_path)
                           success_flags['audio_revision'] = audio_rev_ok
                           if not audio_rev_ok: revision_success = False
                      except Exception as e:
                           print(f"Error during audio_revision for {directory_path}: {e}")
                           success_flags['audio_revision'] = False
                           revision_success = False
                 else:
                      print("Skipping audio revision as prerequisite 'audio_detail_captions.txt' or 'video_consistent_captions.txt' is missing.")
                      success_flags['audio_revision'] = False
                      revision_success = False
            elif run_flags.get('audio_revision', False): # If audio revision requested but video revision failed
                 print("Skipping audio revision due to video revision failure.")
                 success_flags['audio_revision'] = False
                 # revision_success is already False

            success_flags['revision'] = revision_success # Overall success of the revision block


    elif run_flags.get('revision', False):
         print("Skipping revision step due to previous step failure or missing inputs.")
         success_flags['revision'] = False
    else:
         print("Skipping revision step.")
         success_flags['revision'] = True # Mark as success if skipped for dependency chain

    # --- Step 3: Alignment ---
    # Run alignment only if revision step was successful *or skipped* and alignment is requested
    if run_flags.get('alignment', False) and success_flags.get('revision', True):
        # Check for necessary input files from revision step
        video_consistent_path = os.path.join(directory_path, 'video_consistent_captions.txt')
        audio_revised_path = os.path.join(directory_path, 'audio_revised_captions.txt')
        # Also need the video file itself (checked within the function)

        prereqs_met = os.path.exists(video_consistent_path) and os.path.exists(audio_revised_path)

        if prereqs_met:
            try:
                # Call the alignment function from the revision module
                success_flags['alignment'] = revision.perform_av_alignment(directory_path)
            except Exception as e:
                 print(f"Error during AV Alignment for {directory_path}: {e}")
                 success_flags['alignment'] = False
        else:
            print(f"Skipping AV alignment for {directory_path} due to missing prerequisite revised caption files.")
            success_flags['alignment'] = False

    elif run_flags.get('alignment', False):
         print("Skipping AV alignment step due to previous step failure.")
         success_flags['alignment'] = False
    else:
         print("Skipping AV alignment step.")
         success_flags['alignment'] = True # Mark as success if skipped for dependency chain

    # --- Step 4: QA Generation ---
    # Run QA only if alignment step was successful *or skipped* and QA is requested
    if run_flags.get('qa_generation', False) and success_flags.get('alignment', True): # Depends on alignment block success/skip
        # Check for necessary input files (revised captions + alignment file)
        video_consistent_path = os.path.join(directory_path, 'video_consistent_captions.txt')
        audio_revised_path = os.path.join(directory_path, 'audio_revised_captions.txt')
        alignment_caption_path = os.path.join(directory_path, 'av_alignment_captions.txt') # Now depends on alignment step

        prereqs_met = os.path.exists(video_consistent_path) and \
                      os.path.exists(audio_revised_path) and \
                      os.path.exists(alignment_caption_path) # Alignment file is now required

        if prereqs_met:
            # Find the correct metadata for this video_id (logic remains the same)
            this_video_meta = None
            if qa_metadata and isinstance(qa_metadata, list) and all(isinstance(item, dict) for item in qa_metadata):
                 this_video_meta = next((item for item in qa_metadata if item.get('video_id_for_alignment') == video_id), None)

            use_gemini_qa = run_flags.get('use_gemini_qa', True)
            try:
                success_flags['qa_generation'] = qa_generation.generate_advanced_qa(
                    qa_folder_path=directory_path,
                    use_gemini=use_gemini_qa,
                    metadata=[this_video_meta] if this_video_meta else None,
                    file_lock=file_lock
                )
            except Exception as e:
                 print(f"Error during QA Generation for {directory_path}: {e}")
                 success_flags['qa_generation'] = False
        else:
            print(f"Skipping QA generation for {directory_path} due to missing prerequisite revised captions or alignment file.")
            success_flags['qa_generation'] = False

    elif run_flags.get('qa_generation', False):
         print("Skipping QA generation step due to previous step (alignment/revision) failure.")
         success_flags['qa_generation'] = False
    else:
         print("Skipping QA generation step.")
         success_flags['qa_generation'] = True # Mark as success if skipped

    # --- Summary ---
    end_time = time.time()
    duration = end_time - start_time
    print(f"-------------------------------------------")
    print(f"Finished Processing: {directory_path} in {duration:.2f} seconds")
    # Calculate overall success based on requested steps finishing ok
    overall_success = True
    # Define the sequence of dependent blocks
    pipeline_blocks = ['captioning', 'revision', 'alignment', 'qa_generation']
    current_step_success = True
    for block_name in pipeline_blocks:
        block_requested = run_flags.get(block_name, False)
        if block_requested:
            if current_step_success: # Check if the previous block succeeded or was skipped successfully
                # Check individual step flags within the block if necessary (e.g., video/audio captioning)
                # For simplicity, using the block-level success flag calculated earlier
                block_success = success_flags.get(block_name, False)
                if not block_success:
                    overall_success = False
                    current_step_success = False # Mark failure to skip subsequent dependent blocks
            else: # Previous block failed, so this dependent block cannot run successfully
                 overall_success = False
                 # Ensure this block's flag is marked False if it wasn't even attempted
                 if block_name not in success_flags: success_flags[block_name] = False

        # Update current_step_success for the next iteration
        # If block was not requested, treat as success for dependency chain
        # If block was requested, use its actual success status
        if not block_requested:
            current_step_success = current_step_success # Maintain status
        else:
             current_step_success = success_flags.get(block_name, False)


    print(f"Overall Dir Success: {overall_success}")
    print(f"Detailed Success Status: {success_flags}")
    print(f"-------------------------------------------")

    return overall_success


if __name__ == "__main__":
    print("Starting Video/Audio Processing Pipeline...")

    # --- Configuration ---
    # (Keep CSV reading and base_dir setup) ...
    csv_file_path = config.CSV_PATH
    if not os.path.exists(csv_file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_csv_path = os.path.join(script_dir, os.path.basename(csv_file_path)) # Use basename from config
        if os.path.exists(potential_csv_path):
            csv_file_path = potential_csv_path
            print(f"Using CSV file found relative to script: '{csv_file_path}'")
        else:
            print(f"CRITICAL ERROR: CSV file not found at '{config.CSV_PATH}' or relative path '{potential_csv_path}'. Please update config.py or the path.")
            exit()
    else:
         print(f"Using CSV file: '{csv_file_path}'")


    base_dir = config.BASE_DIR
    if not os.path.isdir(base_dir):
        print(f"CRITICAL ERROR: Base directory not found at '{base_dir}'. Please update config.py.")
        exit()

    directory_list = utils.read_directory_list(csv_file_path, base_dir=base_dir)
    if not directory_list:
        print("No directories found to process. Exiting.")
        exit()

    all_metadata = utils.read_metadata(csv_file_path)

    # --- Select Steps to Run ---
    run_pipeline_flags = {
        'captioning': True,      # Master flag for captioning block
        'video_caption': True,   # Generate initial video captions
        'audio_caption': True,   # Generate initial audio captions
        'use_gemini_audio': True, # Use Gemini for audio? (False for DashScope)

        'revision': True,        # Master flag for revision block
        'video_revision': True,  # Run video consistency revision
        'audio_revision': True,  # Run audio revision based on visual context

        'alignment': True,       # Master flag for AV alignment (NEW)

        'qa_generation': True,    # Master flag for QA generation
        'use_gemini_qa': False,   # Use Gemini for QA? (True for Gemini, False for DeepSeek/Volc)

        'aggregate_and_filter': True # Run final aggregation and filtering
    }

    # Optional: Limit the number of directories for testing

    # directory_list = directory_list[:2]
    # print(directory_list)
    # if all_metadata and isinstance(all_metadata, list):
    #     all_metadata = all_metadata[:len(directory_list)] # Slice metadata accordingly

    print(f"Found {len(directory_list)} directories to process based on CSV and base_dir.")
    print(f"Pipeline Steps Enabled: {run_pipeline_flags}")
    print(f"Base Directory: {base_dir}")

    # --- Execution Method ---
    execution_mode = 'processes' # 'sequential', 'threads', 'processes'
    # NOTE: 'threads' or 'processes' might have issues with strict dependencies
    # between revision steps and alignment/QA unless handled carefully.
    # Sequential is safest for correctness with these dependencies.

    results = []

    # (Keep execution logic for sequential, threads, processes)
    # Ensure the call to process_single_directory passes the flags correctly
    if execution_mode == 'sequential':
        print("\n--- Running in Sequential Mode ---")
        for directory in tqdm(directory_list, desc="Processing Sequentially"):
             abs_directory_path = os.path.abspath(directory) # Ensure absolute path
             if os.path.isdir(abs_directory_path):
                 results.append(process_single_directory(abs_directory_path, run_pipeline_flags, qa_metadata=all_metadata))
             else:
                 print(f"Warning: Directory not found, skipping: {abs_directory_path}")
                 results.append(False)

    elif execution_mode == 'threads':
        print("\n--- Running in Thread Pool Mode (Use with caution due to potential step dependencies) ---")
        MAX_WORKERS_THREADS = config.MAX_WORKERS_THREADS
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_THREADS) as executor:
             futures = {}
             for directory in directory_list:
                  abs_directory_path = os.path.abspath(directory) # Ensure absolute path
                  if os.path.isdir(abs_directory_path):
                      futures[executor.submit(process_single_directory, abs_directory_path, run_pipeline_flags, qa_metadata=all_metadata)] = abs_directory_path
                  else:
                      print(f"Warning: Directory not found, skipping submission: {abs_directory_path}")
                      results.append(False)

             for future in tqdm(as_completed(futures), total=len(futures), desc="Processing (Threads)"):
                 directory = futures[future]
                 try:
                     results.append(future.result())
                 except Exception as e:
                     print(f"Error processing directory {directory} in thread: {e}")
                     results.append(False)

    elif execution_mode == 'processes':
        print("\n--- Running in Process Pool Mode (Use with caution due to potential step dependencies and locking) ---")
        MAX_WORKERS_PROCESSES = 1# Use available cores
        if config.MAX_WORKERS_PROCESSES:
            MAX_WORKERS_PROCESSES = config.MAX_WORKERS_PROCESSES
        print(f"Using {MAX_WORKERS_PROCESSES} processes.")
        manager = multiprocessing.Manager()
        file_lock = manager.Lock() # Pass lock if needed (though less likely needed now)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS_PROCESSES) as executor:
            futures = {}
            for directory in directory_list:
                 abs_directory_path = os.path.abspath(directory) # Ensure absolute path
                 if os.path.isdir(abs_directory_path):
                     # Pass lock if QA generation needs it (unlikely if writing per-dir files)
                     futures[executor.submit(process_single_directory, abs_directory_path, run_pipeline_flags, qa_metadata=all_metadata, file_lock=None)] = abs_directory_path
                 else:
                      print(f"Warning: Directory not found, skipping submission: {abs_directory_path}")
                      results.append(False) # Pre-mark as failure

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing (Processes)"):
                directory = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing directory {directory} in process: {e}")
                    results.append(False) # Mark as failure

    # --- Aggregation and Filtering Step ---
    if run_pipeline_flags.get('aggregate_and_filter', False):
        print("\n===========================================")
        print("--- Starting QA Aggregation and Filtering ---")
        print("===========================================")
        # (Keep existing aggregation and filtering logic)
        # Ensure it looks for the correct QA filename based on 'use_gemini_qa' flag
        all_qa_data = []
        collected_files = 0
        skipped_files = 0

        qa_gen_backend_flag = run_pipeline_flags.get('use_gemini_qa', True)
        qa_filename = f"QAs_advance{'_deepseek' if not qa_gen_backend_flag else ''}.json"
        print(f"Looking for QA file: '{qa_filename}' in processed directories...")

        for directory in directory_list:
            abs_directory_path = os.path.abspath(directory) # Ensure absolute path
            if not os.path.isdir(abs_directory_path):
                continue

            qa_file_path = os.path.join(abs_directory_path, qa_filename)
            if os.path.exists(qa_file_path):
                json_content = utils.safe_read(qa_file_path)
                if json_content:
                    try:
                        qa_pairs = json.loads(json_content)
                        if isinstance(qa_pairs, list):
                            if all(isinstance(item, dict) for item in qa_pairs):
                                all_qa_data.extend(qa_pairs)
                                collected_files += 1
                            else:
                                print(f"    Warning: Content in {qa_file_path} is a list but contains non-dictionary items. Skipping.")
                                skipped_files += 1
                        else:
                            print(f"    Warning: Content in {qa_file_path} is not a JSON list. Skipping.")
                            skipped_files += 1
                    except json.JSONDecodeError as e:
                        print(f"    Error: Failed to decode JSON from {qa_file_path}: {e}. Skipping.")
                        skipped_files += 1
                    except Exception as e:
                        print(f"    Error reading or processing {qa_file_path}: {e}. Skipping.")
                        skipped_files += 1
                else:
                    print(f"    Warning: Could not read content from {qa_file_path} (safe_read failed). Skipping.")
                    skipped_files += 1
            else:
                 skipped_files += 1

        print(f"\n--- Aggregation Summary ---")
        print(f"  Directories scanned: {len(directory_list)}")
        print(f"  QA files successfully read and parsed: {collected_files}")
        print(f"  QA files skipped or not found: {skipped_files}")
        print(f"  Total QA pairs collected: {len(all_qa_data)}")

        if all_qa_data:
            consolidated_qa_path = config.CONSOLIDATED_QA_FILENAME
            print(f"\n--- Writing Consolidated QA Data ---")
            print(f"  Saving to: {consolidated_qa_path}")
            write_content = json.dumps(all_qa_data, ensure_ascii=False, indent=4)
            if utils.safe_write(consolidated_qa_path, write_content, mode='w'):
                print(f"  Successfully saved consolidated QA file.")
                print(f"\n--- Filtering Consolidated QA Data ---")
                try:
                    filtered_qa_data = filter_qa_list(all_qa_data, config.BASE_DIR)
                    if filtered_qa_data:
                         filtered_qa_path = config.FILTERED_QA_FILENAME
                         print(f"\n--- Writing Filtered QA Data ---")
                         print(f"  Saving {len(filtered_qa_data)} filtered QA pairs to: {filtered_qa_path}")
                         filtered_write_content = json.dumps(filtered_qa_data, ensure_ascii=False, indent=4)
                         if utils.safe_write(filtered_qa_path, filtered_write_content, mode='w'):
                             print("  Successfully saved filtered QA file.")
                         else:
                              print(f"  Error: Failed to write filtered QA file to {filtered_qa_path}.")
                    else:
                         print("\n--- No QA pairs passed the filter. Filtered file will not be created. ---")
                except Exception as e:
                    print(f"\n--- Error during QA Filtering Process ---")
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Error: Failed to write consolidated QA file to {consolidated_qa_path}. Skipping filtering.")
        else:
            print("\n--- No QA data collected. Skipping consolidation and filtering. ---")
    else:
        print("\n--- Skipping QA Aggregation and Filtering step as per run_flags ---")

    # --- Final Summary ---
    successful_runs = sum(1 for r in results if r is True)
    failed_runs = len(results) - successful_runs
    total_attempted = len(results)
    print("\n===========================================")
    print("Pipeline Execution Summary:")
    print(f"Total directories listed in CSV: {len(directory_list)}")
    print(f"Total directories attempted: {total_attempted}")
    print(f"Successful runs (all requested steps OK): {successful_runs}")
    print(f"Failed/Partial/Skipped runs: {failed_runs}")
    print("===========================================")