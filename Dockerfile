FROM julia:1.6.1

# Use New Zealand mirrors
RUN sed -i 's/archive/nz.archive/' /etc/apt/sources.list

RUN apt update

# Set timezone to Auckland
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y locales tzdata git
RUN locale-gen en_NZ.UTF-8
RUN dpkg-reconfigure locales
RUN echo "Pacific/Auckland" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata
ENV LANG en_NZ.UTF-8
ENV LANGUAGE en_NZ:en

# Create user 'kaimahi' to create a home directory
RUN useradd kaimahi
RUN mkdir -p /home/kaimahi/
RUN chown -R kaimahi:kaimahi /home/kaimahi
ENV HOME /home/kaimahi

# Install apt packages
RUN apt update
RUN apt install -y gettext libcairo2 libpango1.0-0

# Install python + other things
RUN apt update
RUN apt install -y python3-dev python3-pip

# Install julia packages
USER kaimahi
RUN julia -e 'using Pkg; Pkg.add("Pluto")'
RUN julia -e 'using Pkg; Pkg.add([ \
  Pkg.PackageSpec(name="BenchmarkTools", version="1.0.0"), \
  Pkg.PackageSpec(name="CSV", version="0.8.5"), \
  Pkg.PackageSpec(name="Chain", version="0.4.6"), \
  Pkg.PackageSpec(name="DataFrames", version="1.1.1"), \
  Pkg.PackageSpec(name="DifferentialEquations", version="6.17.1"), \
  Pkg.PackageSpec(name="Distributions", version="0.24.18"), \
  Pkg.PackageSpec(name="LaTeXStrings", version="1.2.1"), \
  Pkg.PackageSpec(name="LazyArrays", version="0.21.5"), \
  Pkg.PackageSpec(name="Plots", version="1.16.2"), \
  Pkg.PackageSpec(name="PlutoUI", version="0.7.9"), \
  Pkg.PackageSpec(name="StatsBase", version="0.33.8"), \
  Pkg.PackageSpec(name="StatsPlots", version="0.14.21"), \
  Pkg.PackageSpec(name="Turing", version="0.16.0") \
  ])'

# Install Python
USER root
RUN apt update
RUN python3 -m pip install --upgrade pip
RUN pip3 install wheel

COPY requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt

# Install R package dependencies
RUN apt update && apt install -y \
  libcurl4-openssl-dev \
  libssl-dev \
  libxml2-dev \
  libfontconfig1-dev \
  libmagick++-dev \
  cargo \
  libharfbuzz-dev \
  libfribidi-dev \
  desktop-file-utils \
  libudunits2-dev \
  gdal-bin \
  libzmq3-dev \
  wget

# install pandoc
RUN wget https://github.com/jgm/pandoc/releases/download/2.5/pandoc-2.5-1-amd64.deb -P /root
RUN dpkg -i /root/pandoc-2.5-1-amd64.deb

# Install R
RUN apt install -y r-base

ENV DOWNLOAD_STATIC_LIBV8=1
RUN Rscript -e 'install.packages(c("tidyverse", "devtools", "drake", "bookdown", "kableExtra", "here", "reticulate", "furrr", "optparse", "shiny", "rstan"), repos = "https://cran.stat.auckland.ac.nz", Ncpus=parallel::detectCores()-1, dependencies=TRUE)'
