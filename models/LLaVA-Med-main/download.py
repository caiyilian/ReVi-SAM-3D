from modelscope import snapshot_download 
# model_dir = snapshot_download('microsoft/Phi-3.5-vision-instruct',local_dir='./Phi-3.5-vision-instruct')
# print(f'{model_dir=}')
model_dir = snapshot_download('ecoxial2007/Phi-3.5V-Med',local_dir='./ecoxial2007/Phi-3.5V-Med')
print(f'{model_dir=}')

