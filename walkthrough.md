# Week 4 Notes

## PortScan Detection Limitation
The live pipeline works perfectly for normal (BENIGN) traffic - 100% accuracy on real captured traffic.
For PortScan attacks, we found the training data (CICIDS2017) doesn't match how a real nmap scan looks.
The dataset shows SYN=0, PSH=1 for PortScan flows, but a real nmap -sS scan shows SYN=2, PSH=0.
This was confirmed across all 18,139 PortScan rows in the training data - it's a known issue with 
this dataset, not a bug in our pipeline code.
