# Testing different execution styles for pinpoint

## learnings
- first 3 are almost the same -> benchmark.py and for loop dont really make a difference
- watt calculation considerably lower, might be wrong still?


### through cmd without benchmark.py

	Energy counter stats for 'sleep 10':
	[interval: 50ms, before: 2000ms, after: 2000ms, delay: 0ms, runs: 20]

	238.31 J nvml:nvidia_geforce_gtx_970_0	( +- 0.34% )
	  7.24 J rapl:ram                     	( +- 0.14% )
	 25.71 J rapl:cores                   	( +- 5.38% )
	189.83 J rapl:pkg                     	( +- 0.73% )

	10.00348395 seconds time elapsed ( +- 0.00% )


### through benchmark.py with 1 for loop and r -20

	Energy counter stats for 'sleep 10':
	[interval: 50ms, before: 2000ms, after: 2000ms, delay: 0ms, runs: 20]

	247.67 J nvml:nvidia_geforce_gtx_970_0	( +- 15.60% )
	  7.24 J rapl:ram                     	( +-  0.16% )
	 26.52 J rapl:cores                   	( +-  3.18% )
	188.76 J rapl:pkg                     	( +-  4.19% )

	10.00340289 seconds time elapsed ( +- 0.00% )


### through benchmark.py with for loop but no watts

	sleep_summary Experiment Data:
	Average energy consumption <Joules (deviation %)>:
	GPU: 240.24 (0.36%)
	RAM: 7.27 (0.08%)
	CPU: 26.78 (3.42%)
	PKG: 190.91 (0.48%)
	Total: 274.30
	Average accuracy <accuracy (deviation %)>:
	0.00 (0.00%)
	Efficiency: 0


### through benchmark.py with for loop and watts

	sleep Experiment Data:
	Average energy consumption <Joules (deviation %)>:
	GPU: 170.35 (4.93%)
	RAM: 5.12 (0.30%)
	CPU: 18.10 (2.99%)
	PKG: 134.20 (0.41%)
	Total: 193.57
	Average accuracy <accuracy (deviation %)>:
	0.00 (0.00%)
	Efficiency: 0