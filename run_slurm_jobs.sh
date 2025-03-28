#!/bin/bash
# Usage: bash run_slurm_jobs.sh -s settings -t type -e experiments
echo "Running batch jobs"

# Parse flags
while :; do
    case $1 in
        -s|--settings) # Model settings
            if [ "$2" ]; then
                settings=$2
                shift
            else
                printf 'ERROR: "--settings" requires a non-empty option argument.\n' >&2
                exit 1
            fi
            ;;
        -t|--type) # Model type
            if [ "$2" ]; then
                type=$2
                shift
            else
                printf 'ERROR: "--type" requires a non-empty option argument.\n' >&2
                exit 1
            fi
            ;;
        -e|--experiments) # Experiment numbers
            if [ "$2" ]; then
                IFS=',' read -r -a experiments <<< "$2"
                shift
            else
                printf 'ERROR: "--numbers" requires a non-empty option argument.\n' >&2
                exit 1
            fi
            ;;
        -n|--number-ensemble-members) # Number of ensemble members
            if [ "$2" ]; then
                IFS=',' read -r -a num_members <<< "$2"
                shift
            else
                printf 'ERROR: "--number" requires a non-empty option argument.\n' >&2
                exit 1
            fi
            ;;
        -p|--path)  # Additional path for the wrap command
            if [ "$2" ]; then
                path=$2
                shift
            else
                printf 'ERROR: "--path" requires a non-empty option argument.\n' >&2
                exit 1
            fi
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: If no more options then break out of the loop.
            break
    esac
    shift
done

# Default experiment setting is all of them
if [ -z "$experiments" ]; then
  experiments=(1 2 3 4 5 6 7 8 9 10)
fi

# Default number of ensemble members is all of them
if [ -z "$num_members" ]; then
  num_members=(1 2 4 8 16)
fi

# Run jobs
for i in "${experiments[@]}"; do
  # Set time and gpu memory based on number of members
  for j in "${num_members[@]}"; do
    if [ "$j" == 1 ]; then
      time=300
      gpu_mem=20
    elif [ "$j" == 2 ]; then
      time=10
      gpu_mem=20
    elif [ "$j" == 4 ]; then
      time=300
      gpu_mem=20
    elif [ "$j" == 8 ]; then
      time=336
      gpu_mem=60
    elif [ "$j" == 16 ]; then
      # Deep Ensemble needs more memory and time
      if [ "$type" == "Deep_Ensemble" ]; then
        time=336
        gpu_mem=60
      else
        time=96
        gpu_mem=60
      fi
    fi

    # Construct output and wrap
    output="storage/logs/"$type"_ViT_base_32_"$j"_members_"$settings$i".out"
    wrap="batch_script.slurm $settings$i.json $type $j"
    if [ -n "$path" ]; then
        wrap="$wrap  $path"
    fi
    job_name="${type:0:1}""m"$j"s"$i

    slurm_call="sbatch -J "$job_name" --time=1-"$time" --mem-per-cpu=32g --gpus=4 --gres=gpumem:"$gpu_mem"g --output=$output $wrap"

    # Run job
    echo "Running job with settings $settings$i and type $type with $j members"

    eval "$slurm_call"
    sleep 2
  done
done        
