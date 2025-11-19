#!/bin/bash
# EBE.sh: a single event
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd  # Should be /srv/scratch by default
echo

#set -x  # Enable debugging

echo "Activating conda environment..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate /usr/conda/ape

# Export osu-hydro and trento binary location to path
export PATH=/usr/bin:$PATH

# Run the script
echo "Running event..."
python /usr/ape/ebe_ape.py
echo "Event complete!"

# Transfer relevant output files to /srv
echo "Moving results parquet to /srv..."
mv /srv/results/*/*.pickle /srv
mv /srv/results/*/*.dat /srv
mv /srv/results/*/*.npy /srv
mv /srv/results/*/*.nc /srv

echo "Job complete! Have a great day! :)"
