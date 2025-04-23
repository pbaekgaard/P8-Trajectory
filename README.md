# P8-Trajectory ft. Tianyi Li

## Getting the Data

Use the data-tools.py tool to get the data.

### Usage

- Download T-Drive: `./data-tools.py download`
- Preprocessing: `./data-tools.py preprocess`
- Additionally you can do both at the same time: `./data-tools.py download preprocess`

You can also limit preprocessing to specific ones:
`./data-tools.py preprocess --only=timestamporder,deduplication`

- Install OSTC using `pip3 install ./OSTC`


on server: 
- ``git clone https://github.com/pbaekgaard/P8-Trajectory --recurse-submodules``
- Install CMAKe on server using `sudo apt install cmake`
- Install ostc on server using `pip3 install ./OSTC`