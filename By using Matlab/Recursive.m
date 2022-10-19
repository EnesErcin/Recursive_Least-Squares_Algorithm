data = [1,4,5,8,9,10,11,12,9,8,7];
fs = 8000;
t = 0:1:length(data)-1;
t = t/fs;
x =randn(1,length(data));
n=filter([0 0.5],1,x);
d = data + n ;
mu = 0.01;
filt_dim = 3;
w = zeros(1,filt_dim);
y = zeros(1,length(data));
e = y ;


% Apply adaptive filter
for m = (filt_dim+1):1:(length(t)-1)
    sum = 0;
    for i = 1:1:filt_dim
        sum = sum + w(i)*x(m-1);
    end
    y(m) = sum;
    e(m) = data(m)-y(m);
    for i=1:1:filt_dim
        w(i)= w(i)+2*mu*e(m)*x(m-i);
    end
end

%Plotting for input,noise,corrpted_noise and ;
subplot(4,1,1),plot(data);grid;ylabel("Original Speech");
subplot(4,1,2),plot(d);grid;ylabel("Corrpt Speech");
subplot(4,1,3),plot(x);grid;ylabel("Reference Noise");
subplot(4,1,4),plot(e);grid;ylabel("Clean Speech");
xlabel("Number of samples");


%Since there is no nonlinear section all weigth became equal ??
% w(i)= w(i)+2*mu*e(m)*x(m-i) --> last terms make it nonlinear 
% Fixed from m-1 to m-i

%Freqauncy domain effect

%Single sided amplitdue specturm
DATA = 2*abs(fft(data))/length(data); DATA(1) = DATA(1)/2;
D=2*abs(fft(d))/length(d); d(1) = d(1)/2;
E = 2*abs(fft(e))/length(e); E(1)= E(1)/2;
f= [0:1:length(data)/2]*fs/length(data);

%Plot Signals
figure
subplot(3,1,1),plot(f,DATA(1:length(f)));grid;ylabel("Original Spectrum");
subplot(3,1,2),plot(f,D(1:length(f)));grid;ylabel("Corrpt Specturm");
subplot(3,1,3),plot(f,E(1:length(f)));grid;ylabel("Clean Spectrum");
xlabel("Frequancy Hz");





