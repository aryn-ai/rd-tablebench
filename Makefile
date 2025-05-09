
help:
	@echo "Targets"
	@echo "    data: downloads the data"

data:
	-mkdir data
	wget -P data https://huggingface.co/datasets/reducto/rd-tablebench/resolve/main/rd-tablebench.zip
	cd data && unzip rd-tablebench.zip
