import shutil
import sys
from logparser import Drain, Brain

def parse_logs():
    """
    Parses HDFS log files using either the Drain or Brain parsing engine.

    The function sets up the parsing engines and input/output directories, and then parses each log file
    using each parsing engine. The output is saved to the specified output directory.

    Args:
        None

    Returns:
        None
    """
    print("setup")
    parsing_engines = ["brain", "drain"]
    dataset    = 'HDFS'
    input_dir  = '../../../../dataset/hdfs/'  # The input directory of log file
    output_dir = '../../../../logs/'  # The output directory of parsing results
    log_files   = ['HDFS.log'] #['HDFS_2K.log', 'HDFS_10K.log', 'HDFS_50K.log', 'HDFS_100K.log','HDFS_500K.log' ]  # The input log file name
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format

    # Regular expression list for optional preprocessing (default: [])
    regex      = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]

    print("housekeeping")
    shutil.rmtree(output_dir, ignore_errors=True)  # Clear output directory
    for log_file in log_files:
        print("parsing")
        for parsing in parsing_engines:
            log_dir = output_dir + dataset + '/' + parsing + '/'
            if parsing == "drain":
                threshold         = 0.5  # Similarity threshold
                depth      = 4  # Depth of all leaf nodes
                parser = Drain.LogParser(log_format, indir=input_dir, outdir=log_dir,  depth=depth, st=threshold, rex=regex)
            elif parsing == "brain":
                threshold  = 2  # Similarity threshold
                delimeter  = []  # Depth of all leaf nodes
                parser = Brain.LogParser(dataset, log_format, input_dir, log_dir, threshold=threshold,delimeter=delimeter, rex=regex)    
            else:
                print("invalid parsing method")
                sys.exit(1)             
            parser.parse(log_file)

if __name__ == "__main__":
    parse_logs()