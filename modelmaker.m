data = load('data/solarsample2016.csv');
ghi = reshape(data(:,6), [24 365])';
dailyData = sum(ghi, 2)/1000;

mean(dailyData)
modelFreq = fft(dailyData) .* [1;1;zeros(362,1);1];
[modelFreq(end);modelFreq(1);modelFreq(2)]/365
dailyModel = real(ifft(modelFreq));

plot(1:365, dailyData, '*')
hold on
plot(1:365, dailyModel, 'r');
save('data/dailyInsolation.csv', 'dailyModel')
