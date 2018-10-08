import os

_writer = open('./baselines/bba+.log','w')
for p in os.listdir('./bba_results'):
    _file = open('./bba_results/' + p, 'r')
    _past_bitrate = 0.
    _total_smoothness = 0.
    _total_bitrate = 0.
    _total_rebuf = 0.
    for _line in _file:
        _lines = _line.split()
        if len(_lines) > 0:
            _bitrate = float(_lines[1])
            _rebuf = float(_lines[3])
            _total_smoothness += abs(_bitrate - _past_bitrate)
            _total_bitrate += _bitrate
            _total_rebuf += _rebuf
            _past_bitrate = _bitrate
    _writer.write(str(_total_bitrate * 198. / 48.) + ' ' + str(_total_rebuf) + ' ' + str(_total_smoothness) + '\n')
    _file.close()
_writer.close()

os.system('python2 cdfBitrate.py')
os.system('python2 cdfRbf.py')
os.system('python2 cdfsmt.py')
os.system('python2 cdfQoe.py')
