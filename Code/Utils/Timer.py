import time

class Timer:

	def __init__(self):
		self.timer_dict={}

	def start_timer(self,name):
		self.timer_dict[name]=[time.time(),None,None]

	def stop_timer(self,name,do_print=True):
		curr_time = time.time()
		diff_time = curr_time - self.timer_dict[name][0]
		self.timer_dict[name][1] = curr_time
		self.timer_dict[name][2] = diff_time
		if do_print:
			print(f'TIMER: {name:20s} {diff_time:.20f}')
		return diff_time

	def get_timers(self):
		return self.timer_dict
