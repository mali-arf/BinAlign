Here is the replication package to conduct experiments on BinAlign.
-d <dataset>
-g <GPU_ID>
-b <batch size>
-e <number of epochs>
-l <number of instructions>
-k <fold for cross validation>

# to run BinAlign experiments on the whole binaries
python3 o-glassesX-Padding.py -d complete-Win-dataset/ -g 0 -s 1000 -b 64 -e 20 -l 64 -k 4

# to run BinAlign experiments on the text sections of the binaries
python3 o-glassesX-Padding.py -d win-text-for-oglass/ -g 3 -s 1000 -b 64 -e 20 -l 64 -k 4

# to run integrated (o-glassesX + BinAlign) experiments on the text sections of the binaries
python3 o-glassesX-BinAlign-Combined.py -d win-text-for-oglass/  -g 2 -s 1000 -b 64 -e 20 -l 64 -k 4

# to run integrated (o-glassesX + BinAlign) experiments on whole binaries
python3 o-glassesX-BinAlign-Combined.py -d complete-Win-dataset/ -g 0 -s 1000 -b 64 -e 20 -l 64 -k 4

# to run o-glassesX experiments on the text sections of the binaries
python3 o-glassesX-for-win.py -d win-text-for-oglass/ -g 3 -s 1000 -b 64 -e 20 -l 64 -k 4
