FROM dragonflyscience/dragonverse-18.04:latest

RUN apt update

# Install python + other things
RUN apt install -y python3-dev python3-pip

COPY requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt
