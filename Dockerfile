FROM rocker/tidyverse

ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_NZ.UTF-8
ENV LANGUAGE en_NZ:en

# Set New Zealand mirrors and set timezone to Auckland
RUN sed -i 's/archive/nz.archive/' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y tzdata
RUN echo "Pacific/Auckland" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Set the locale to New Zealand
RUN apt-get -y install locales
RUN locale-gen en_NZ.UTF-8

RUN dpkg-reconfigure locales

RUN apt-get update && \
  apt upgrade --yes && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN apt update && apt install -y libxt-dev

RUN Rscript -e 'install.packages("rstan")'
RUN Rscript -e 'install.packages("wbstats")'
RUN Rscript -e 'install.packages("here")'
RUN Rscript -e 'install.packages("furrr")'
RUN Rscript -e 'install.packages("remotes")'
RUN Rscript -e 'remotes::install_github("GuangchuangYu/nCov2019")'
RUN Rscript -e 'install.packages("prettydoc")'
