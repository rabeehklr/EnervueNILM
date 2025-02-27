import os

def merge_docker_files(root_directory, output_file):
    """
    Recursively read contents of all files in a directory and merge them into a single output file.
    
    :param root_directory: Path to the root directory of the Docker setup
    :param output_file: Path to the output text file where contents will be merged
    """
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through all directories and files
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for filename in filenames:
                # Create full file path
                full_path = os.path.join(dirpath, filename)
                
                # Skip binary or very large files
                try:
                    # Check file size (skip files larger than 1MB)
                    if os.path.getsize(full_path) > 1024 * 1024:
                        outfile.write(f"# SKIPPED: {full_path} (File too large)\n\n")
                        continue
                    
                    # Read file contents
                    with open(full_path, 'r', encoding='utf-8') as infile:
                        # Write file path and name as a comment
                        outfile.write(f"# File: {full_path}\n")
                        outfile.write(f"# Relative Path: {os.path.relpath(full_path, root_directory)}\n")
                        outfile.write("# ---\n")
                        
                        # Write file contents
                        outfile.write(infile.read())
                        
                        # Add separators between files
                        outfile.write("\n\n# ----------------------------------------\n\n")
                
                except UnicodeDecodeError:
                    # Handle potential encoding issues
                    outfile.write(f"# SKIPPED: {full_path} (Unable to read file encoding)\n\n")
                except PermissionError:
                    # Handle permission denied errors
                    outfile.write(f"# SKIPPED: {full_path} (Permission denied)\n\n")

def main():
    # Example usage
    root_dir = r"C:\Users\ASUS\Desktop\S8 PROJECT\DockerSetup"  # Replace with your Docker setup directory
    output_path = 'docker_files_merged.txt'
    
    merge_docker_files(root_dir, output_path)
    print(f"All files merged into {output_path}")
#docker exec -it dockersetup-db-1 psql -U postgres -d power_consumption -c "\COPY (SELECT * FROM aggregate_data) TO '/tmp/aggregate_data.csv' WITH CSV HEADER"

#docker cp dockersetup-db-1:/tmp/aggregate_data.csv ./aggregate_data.csv
if __name__ == "__main__":
    main()