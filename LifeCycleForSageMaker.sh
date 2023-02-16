set -e
ENVIROMENT=python3
FILE="/home/ec2-user/SageMaker/test-s3.ipynb"
source /home/ec2-user/anaconda3/bin/activate "$ENVIROMENT"
pip install tensorflow
nohup jupyter nbconvert $FILE --ExecutePreprocessor.kernel_name=python3 --to notebook --inplace --ExecutePreprocessor.timeout=7200 --execute &
source /home/ec2-user/anaconda3/bin/deactivate
IDLE_TIME=7200
echo "autostop script"
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py
echo "empezando autostop en cron"
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/bin/python3 $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab
