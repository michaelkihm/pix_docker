FROM python:3.7-slim-stretch

RUN apt update -y && apt install build-essential -y &&  apt-get install manpages-dev -y
RUN pip install --upgrade pip
RUN pip install pixplot

COPY entrypoint.sh /root

ENTRYPOINT [ "sh" ,"/root/entrypoint.sh"]