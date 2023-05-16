# KMP-Cuda
**Our final report is in the Report.pdf**

In this project we have tried to optimize the KMP algorithm on the GPU using CUDA. We have implemented the optimizations of two algorithms: the naive KMP algorithm and the DFA algorithm. We have also explored three optimizations that focus on improving the total runtime. The best kernel we have got is 2.5x faster than the baseline kernel, utilizing memory coalescing. We have also cut down the total execution time by a factor of 1.58 using unified memory.
<!-- 
# Status( May 8, 2023):
* State machine kernel updated on branch xye16/state_machine.
* With 1G text, the runtime of the kernel is only ~2.75ms, while the total time including cudaMemcpy is ~45ms. 
* Next steps, experiment with parameters to get the optimal performance. -->

<!-- # Meeting 1 Overview:
Last week, we implemented the serial code for KMP and ran it on a basic dataset, on the RAI server. Currently, we are working on the basic kernel for KMP. In the following weeks, we aim to improve the performance by making dynamic use of shared and constant memory for Longest Prefix Suffix table (size of the table decides the type of memory), and use of streams to overlap data loading and computations. Also, we would work on preparing a more comprehensive dataset to give better performance metrics.
 -->

# Group Member
* Feiran Wang(feiranw2)
* Aryan Raja(aryankr2)
* Xingjian Ye(xye16)

