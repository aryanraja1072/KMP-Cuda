# KMP-Cuda
Status( May 8, 2023):
* State machine kernel updated on branch xye16/state_machine.
* With 1G text, the runtime of the kernel is only ~2.75ms, while the total time including cudaMemcpy is ~45ms. 
* Next steps, experiment with parameters to get the optimal performance.

# Meeting 1 Overview:
Last week, we implemented the serial code for KMP and ran it on a basic dataset, on the RAI server. Currently, we are working on the basic kernel for KMP. In the following weeks, we aim to improve the performance by making dynamic use of shared and constant memory for Longest Prefix Suffix table (size of the table decides the type of memory), and use of streams to overlap data loading and computations. Also, we would work on preparing a more comprehensive dataset to give better performance metrics.


# Group Member
* Feiran Wang(feiranw2)
* Aryan Raja(aryankr2)
* Xingjian Ye(xye16)


Feedback 1:
Hi Feiran, Xingjian and Aryan,

Thank you for your interest! Previously we worked on accelerating regular expression matching as the ECE 508 project when I took this course. You can check https://uillinoisedu-my.sharepoint.com/:w:/g/personal/kunwu2_illinois_edu/Ea1Df63NPWdMh_N-t9y2W9wBFHEWxxEcq6YZdOyzjCPozw?e=ynnKRG to find more elaboration about the motivation behind it.

KMP is aligned with this direction and I believe it is an essential workload. techniques used in implementing BFS could be relevant as it is dealing with transitions. You can check https://algs4.cs.princeton.edu/lectures/keynote/53SubstringSearch-2x2.pdf, e.g., page 20, for illustration to get better intuition of the KMP algorithm.

I believe one important application of string operations are in datatable. For example, log anomaly detection can be seen as identifying the problematic row where each row in the data table is a log entry from the system. Dealing with a datatable may provide massive input and a dimension of parallelism. You can get some dataset to work on from repositories like logpai/loghub: A large collection of system log datasets for log analysis research (github.com).

These are the thoughts I have for now. Please feel free to ask us any questions.
