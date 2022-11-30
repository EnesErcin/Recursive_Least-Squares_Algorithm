load('RLS_DATA.mat');
delay = finddelay(TX_I,RX_I);
period = 35;

hold off;
clf('reset');
refreshdata;

for per = 100:102
    txt = sprintf('Word: %f ', length(TX_Q(per*(period)+1:(1+per)*period+1)));
    fprintf(txt);
    figure(1);
    title("First");
    plot(TX_Q(per*(period)+1:(1+per)*period+1),RX_Q(delay+per*(period)+1:delay+(1+per)*period+1));
    hold on
end
grid;

hold off

for per = 5:7
    txt = sprintf('Word: %f ', length(TX_Q(per*(period)+1:(1+per)*period+1)));
    fprintf(txt);
    figure(2);
    title("Second");
    plot(TX_Q(per*(period)+1:(1+per)*period+1),RX_Q(delay+per*(period)+1:delay+(1+per)*period+1));
    hold on
end
grid;