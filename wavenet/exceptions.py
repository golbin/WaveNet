class InputSizeError(Exception):
    def __init__(self, input_size, receptive_fields, output_size):

        message = 'Input size has to be larger than receptive_fields\n'
        message += 'Input size: {0}, Receptive fields size: {1}, Output size: {2}'.format(
            input_size, receptive_fields, output_size)

        super(InputSizeError, self).__init__(message)
