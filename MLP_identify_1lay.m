clear all;
close all;
run dane/zad1_dane_we;
run dane/zad1_dane_test;

%% Using MLP Neural Network to identify simple dynamic linear systems (P,I,D and combinations)
% 1 hidden layer network
% Comments in Polish
% Author: Pawe³ Mêdyk

% Wykorzystanie sieci MLP od identyfikacji podstawowych cz³onów dynamiki
% (P,I,D i ich kombinacji)
% 1 warstwa ukryta
% Imiê i nazwisko autora: Pawe³ Mêdyk

%% konstruowanie bazy uczacej
baza_ucz_we=dane_we';   % baza ucz. - wejscia
% interpretacja wyjsc - [P I D] - rozpoznanie z prawdopodobienstwem od 0 do 1 
baza_ucz_wy=[           % baza ucz. - wyjscia 
    1 0 0;
    1 0 0;
    1 0 0;
    0 1 0;
    0 1 0;
    0 1 0;
    0 0 1;
    0 0 1;
    0 0 1;]'; 

%% konstruowanie bazy testowej
baza_test_we=dane_test';  % baza test. - wejscia
baza_test_wy=[
    1 1 0;
    1 0 1;
    1 1 1;
    1 0 0;
    0 1 0;
    0 0 1;]';

%% struktura sieci
n = size(baza_ucz_we,1);  % liczba wejsc/neronow w warstwie wejsciowej
k1 = 2;                  % liczba neuronow w pierwszej warstwie ukrytej
k2 = size(baza_ucz_wy,1); % liczba neuronow w warstwie wyjsciowej

%% parametry sieci
eta= zeros(2,1);
beta=zeros(2,1);
eta(1) = 0.10;      % wspolczynnik uczenia sie warstwy ukrytej
eta(2) = 0.05;      % wspolczynnik uczenia sie warstwy wyj.
beta(1) = 1.1;      % wspolczynnik stromosci funkcji aktywacji - warstwa ukryta I
beta(2) = 2.1;      % wsp. stromosci f.akt. - warstwa wyj.    
Epoki = 80000;      % liczba epok

%% liczba instancji
lowest_MSE=10^6;    % inicjalizuj najmniejszy MSE duza wartoscia
N=1                 % liczba iteracji/powtorzen eksperymentu; N instancji sieci
%% petla glowna - generuj nowa instancje sieci N razy
for i=1:N
%% inicjowanie macierzy wag
a = -0.5; % dolna granica inicjowania wag
b = 0.5;  % górna granica inicjowania wag
W1 = (b-a)*rand(n+1,k1)+a;  % macierz wag dla pierwszej warstwy ukrytej
W2 = (b-a)*rand(k1+1,k2)+a; % macierz wag dla warstwy wyjsciowej

%% petla uczaca
for ep = 1: Epoki
   L = randi([1, size(baza_ucz_we,2)],1);   % losuj zestaw uczacy
   y0=baza_ucz_we(:,L);             % wyjscie warstwy wejsciowej
   x1 = [-1; y0];                   % zbuduj wektor wejsc I w.ukrytej (dodanie wej. bias)
   u1=W1'*x1;                       % wyznacz wektor pobudzen pierwszej warstwy ukrytej
   y1=1./(1+exp(-beta(1)*u1));      % wyjscie I warstwy ukrytej;
                % funkcja aktywacji pierwszej warstwy ukrytej - sigmoidalna unipolarna
                
   x2=[-1; y1];                     % zbuduj wektor wejsc dla warstwy wyjsciowej (dodanie wej. bias)
   u2=W2'*x2;                     	% wyznacz wektor pobudzen warstwy wyjsciowej
   y2=2*1./(1+exp(-beta(2)*u2))-1;  % wyjscie warstwy wyjsciowej; 
                % funkcja aktywacji warstwy wyjsciowej - sigmoidalnabipolarna
   
   ty = baza_ucz_wy(:,L);           % oczekiwany wektor wyjscia dla wektora uczacego
   d2=ty-y2;                        % wektor bledu pomiedzy wyjsciem sieci, a wyjsciem oczekiwanym
   df2=beta(2)/2*(1-y2.*y2);        % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.wyj
   dW2=eta(2)*x2*(d2.*df2)';        % macierz poprawek dla warstwy wyjsciowej
   W2=W2+dW2;                       % aktualizacja macierzy wag warstwy wyjsciowej

   d1=W2*d2;                % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df1=x2.*(1-x2);          % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.ukryta
   dd1=(d1.*df1)';          % wektor wspolczynnikow bledu
   dd1=dd1(2:end);          % skrocenie wektora wspolczynnikow bledu 
                          % nie mozna poprawiac wektora wag dla
                          % wejscia bias, bo ma ono stala wartosc
   dW1=eta(1)*x1*dd1;       % macierz poprawek dla warstwy wyjsciowej; wsteczna propagacja bledu
   W1=W1+dW1;               % aktualizacja macierzy wag warstwy ukrytej
    
end
end


%% prezentacja najlepszego wypracowanego rozwiazania
Y_matrix=zeros(size(baza_ucz_wy,1),size(baza_test_we,2));
for przykl = 1 : size(baza_test_we,2)  % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie I w.ukrytej
    u1=W1'*x1;                          % pobudzenie I w.ukrytej
    y1=1./(1+exp(-beta(1)*u1));         % wyjscie I w.ukrytej    
    x2=[-1; y1];                        % wejscie w.wyjsciowej
    u2=W2'*x2;                          % pobudzenie w.wyjsciowej
    y2=2*1./(1+exp(-beta(2)*u2))-1;     % wyjscie w.wyjsciowej
    Y_matrix(:,przykl)=y2;              % zapisz wynik dla tego przykladu w macierzy
    plot(t,baza_test_we(:,przykl)); hold on;
    legendInfo{przykl}=sprintf('test %d',przykl);
end
Y_matrix
MSE=immse(Y_matrix,baza_test_wy)

legend(legendInfo);