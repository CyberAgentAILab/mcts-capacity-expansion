FROM ubuntu:16.04

# setup pyenv
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install \
            git \
            make \
            cmake \
            build-essential \
            python-dev \
            python-pip \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            liblzma-dev \
            curl
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN git clone https://github.com/yyuu/pyenv-virtualenv.git $HOME/.pyenv/plugins/pyenv-virtualenv
RUN pyenv install 3.6.0
RUN pyenv global 3.6.0

# install python libraries
RUN pip install --upgrade setuptools
RUN pip install --upgrade pip
RUN pip install gurobipy numpy pandas

# COPY
RUN mkdir $HOME/mcts-capacity-expansion
COPY ./ $HOME/mcts-capacity-expansion
WORKDIR $HOME/mcts-capacity-expansion
