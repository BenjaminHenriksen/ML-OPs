import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
model2 = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)


# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#    for i in range(10):
#         model(inputs)
#         prof.step()

from torch.profiler import profile, tensorboard_trace_handler

with profile(activities=[ProfilerActivity.CPU],record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
    for i in range(3):
        model(inputs)
        prof.step()


with profile(activities=[ProfilerActivity.CPU],record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof2:
    for i in range(3):
        model2(inputs)
        prof2.step()

        
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))
#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_memory_usage", row_limit=30))

#prof.export_chrome_trace("trace_loop.json")