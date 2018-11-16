From python:3.6.7-stretch

RUN apt-get update
RUN pip install numpy scipy
RUN apt-get install -y libhdf5-cpp-100 cifs-utils
RUN mkdir /NAS
RUN mkdir /home/rbry

COPY pythonDLLCPU.so /usr/local/lib/
COPY corrLib.py /usr/local/lib/python3.6/site-packages/
COPY run_command.sh /usr/bin/
COPY dummy_processor.py /home/rbry/
RUN chmod +x /usr/bin/run_command.sh

ENTRYPOINT ["/usr/bin/run_command.sh"]
CMD [""]
