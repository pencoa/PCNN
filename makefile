download:
	wget "https://www.dropbox.com/s/nefg60ripokkuvi/origin_data.zip?dl=0" -O ./data/origin_data.zip
	unzip ./data/origin_data.zip -d ./data

install:
	pip install -r requirements.txt

run:
	python build_data.py
	python train.py
	python evaluate.py
