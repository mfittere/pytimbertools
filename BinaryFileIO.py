#! /bin/env python

import struct
import sys

class BinaryWriter():

    def __init__(self, file_name= None, append= False):
        self.file_name= file_name
        self.f= None

        if file_name is not None:
            if append:
                self.f = open(self.file_name, 'ab')
            else:
                self.f = open(self.file_name, 'wb')
        else:
            self.f= sys.stdout

    def __del__(self):
        if self.f is not None:
            self.f.close()

    def write(self, data, struct_format):
        if self.f is None:
            return

        try:
            data[0]
        except TypeError: # Not iterable, i.e. a scalar
            self.f.write(struct.pack(struct_format, data))
        else:
            for v in data:
                self.f.write(struct.pack(struct_format, v))

    def write_byte(self, data): # int8_t
        return self.write(data, 'b')

    def write_short(self, data): # int16_t
        return self.write(data, 'h')

    def write_int(self, data): # int32_t
        return self.write(data, 'i')

    def write_long(self, data): # int64_t
        return self.write(data, 'q')

    def write_float(self, data): #float
        return self.write(data, 'f')

    def write_double(self, data): #double
        return self.write(data, 'd')


class BinaryReader():

    def __init__(self, file_name= None):
        self.file_name= file_name
        self.f= None
        self.eof= False

        if file_name is not None:
            self.f = open(self.file_name, 'rb')
        else:
            self.f= sys.stdin

    def __del__(self):
        if self.f is not None:
            self.f.close()

    def read(self, length, struct_format, data_size):
        if self.f is None or self.eof:
            return None

        v= [0]*length
        for i in range(length):
            data= self.f.read(data_size)
            if len(data) == data_size:
                v[i]= struct.unpack(struct_format, data)[0]
            else:
                self.eof= True
                raise IOError('No more data to read')
                return None

        if length > 1:
            return v
        else:
            return v[0]

    def seek(self, offset):
        self.f.seek(offset)
        self.eof= False

    def tell(self):
        return self.f.tell()

    def read_byte(self, length): # int8_t
        return self.read(length, 'b', 1)

    def read_short(self, length): # int16_t
        return self.read(length, 'h', 2)

    def read_int(self, length): # int32_t
        return self.read(length, 'i', 4)

    def read_long(self, length): # int64_t
        return self.read(length, 'q', 8)

    def read_float(self, length): #float
        return self.read(length, 'f', 4)

    def read_double(self, length): #double
        return self.read(length, 'd', 8)

if __name__ == '__main__':

    pass
