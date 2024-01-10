input_file_path = 'cleansed_once.txt'
output_file_path = 'output_file.csv'

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


'''
        SubmitTime,WaitTime,RunTime,NProcs,AverageCPUTimeUsed,Used Memory,ReqNProcs,ReqTime,ReqMemory,Status,UserID,GroupID,ExecutableID,QueueID,PartitionID,OrigSiteID,LastRunSiteID
        # 1  JobID		counter
# 2  SubmitTime		in seconds, starting from zero
# 3  WaitTime		in seconds
# 4  RunTime 		runtime measured in wallclock seconds
# 5  NProcs		number of allocated processors
# 6  AverageCPUTimeUsed	average of CPU time over all allocated processors
# 7  Used Memory	average per processor in kilobytes
# 8  ReqNProcs		requested number of processors
# 9  ReqTime: 		requested time measured in wallclock seconds
# 10 ReqMemory		requested memory (average per processor)
# 11 Status		job completed = 1, job failed = 0, job cancelled = 5
# 12 UserID		string identifier for user
# 13 GroupID		string identifier for group user belongs to
# 14 ExecutableID	name of executable
# 15 QueueID		string identifier for queue
# 16 PartitionID	string identifier for partition
# 17 OrigSiteID		string identifier for submission site
# 18 LastRunSiteID	string identifier for execution site
'''