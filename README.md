# gptsne
Rough usage:
(from the src/ directory): python3 -m gptsne.gptsne_mo --help
e.g. python3 -m gptsne.gptsne_mo -d COIL20 --dir "datasets/"

* Datasets used in the paper are in datasets/
* Add your own datasets in csv format, with a header line:
Header: classPosition,#features,#classes,seperator. e.g. classLast,1024,20,comma (from COIL20.data)
