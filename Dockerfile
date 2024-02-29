FROM triton_baichuan_base:23.10.1

COPY boostrap boostrap
COPY bridge_code bridge_code
COPY triton_code triton_code
COPY stress stress