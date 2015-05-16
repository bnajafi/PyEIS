MAJOR = 0
MINOR = 4
MICRO = '1'

dev_labels = ['dev', 'alpha', 'beta']

if isinstance(MICRO, str):
    if MICRO in dev_labels:
        version = '{:d}.{:d}{:s}'.format(MAJOR, MINOR, MICRO)
    elif MICRO not in dev_labels:
        version = '{:d}.{:d}.{:s}'.format(MAJOR, MINOR, MICRO)

elif isinstance(MICRO, int):
    version = '{:d}.{:d}.{:d}'.format(MAJOR, MINOR, MICRO)

