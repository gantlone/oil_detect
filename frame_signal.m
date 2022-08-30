function [Frame_signal] = frame_signal(signal, stride, n)

T = length(signal);
L = floor((T-n-1)/stride);
Frame_signal = cell(L+1,1);
for i=0:L
  framedSignal = signal(i*stride+1:i*stride+n);
  Frame_signal(i+1,1) = {framedSignal};
end
