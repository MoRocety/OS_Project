input_file_path = 'raw_dataset.txt'
output_file_path = 'full_cleaned_dataset.csv'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    output_file.write("SubmitTime,WaitTime,RunTime,NProcs,AverageCPUTimeUsed,Used Memory,ReqNProcs,ReqTime,ReqMemory,Status,UserID,GroupID,ExecutableID,QueueID,PartitionID,OrigSiteID,LastRunSiteID\n")

    for line in input_file:
        # Split the line by tabs
        columns = line.strip().split('\t')

        selected_columns = columns[1:18]
        
        if any(entry == '-1' for entry in selected_columns):
            continue

        # Join the selected columns with commas and write to the output file
        output_line = ','.join(selected_columns)
        output_file.write(output_line + '\n')

print(f"Conversion complete. Output written to {output_file_path}")