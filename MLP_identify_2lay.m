%% Using MLP Neural Network to identify simple dynamic linear systems (P,I,D and combinations)
% 2 hidden layers network
% Comments in Polish
% Author: Pawe³ Mêdyk

% Wykorzystanie sieci MLP od identyfikacji podstawowych cz³onów dynamiki
% (P,I,D i ich kombinacji)
% 2 warstwy ukryte
% Imiê i nazwisko autora: Pawe³ Mêdyk

%% SIEC JEST PRZEUCZANA
% wyczysc pamiec podreczna, zamknij okna, wczytaj dane
close all;
clear all;
run dane/zad1_dane_we;
run dane/zad1_dane_test;

%% konstruowanie bazy uczacej
baza_ucz_we=dane_we';   % baza ucz. - wejscia
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
n = size(baza_ucz_we,1);  % liczba neronow/wejsc w warstwie wejsciowej
k1 = 6;                   % liczba neuronow w pierwszej warstwie ukrytej
k2 = 6;                   % liczba neuronow w drugiej warstwie ukrytej
k3 = size(baza_ucz_wy,1); % liczba neuronow w warstwie wyjsciowej

%% parametry sieci
eta1 = 0.15; % wspolczynnik uczenia sie I warstwy ukrytej 
eta2 = 0.15; % wspolczynnik uczenia sie II warstwy ukrytej
eta3 = 0.05; % wspolczynnik uczenia sie warstwy wyj
beta1 = 1.25;    % wspolczynnik stromosci funkcji aktywacji I w.ukr.
beta2 = 1.25;    % wspolczynnik stromosci funkcji aktywacji II w.ukr.
beta3 = 1.5;     % wspolczynnik stromosci funkcji aktywacji w.wyj.
Epoki = 600000;  % liczba epok

%% liczba instancji
lowest_MSE=10^7 % inicjalizuj najmniejszy MSE duza wartoscia
N=1            % liczba iteracji/powtorzen eksperymentu
%% petla glowna - generuj nowa instancje sieci N razy
for i=1:N
%% inicjowanie macierzy wag
a = -0.5; % dolna granica inicjowania wag
b = 0.5;  % górna granica inicjowania wag
W1 = (b-a)*rand(n+1,k1)+a;  % macierz wag dla pierwszej warstwy ukrytej
W2 = (b-a)*rand(k1+1,k2)+a; % macierz wag dla warstwy wyjsciowej
W3 = (b-a)*rand(k2+1,k3)+a; % macierz wag dla warstwy wyjsciowej

%% petla uczaca
for ep = 1: Epoki
   L = randi([1, size(baza_ucz_we,2)],1);   % losuj zestaw uczacy
   y0=baza_ucz_we(:,L);             % wyjscie warstwy wejsciowej
   
   x1 = [-1; y0];                   % zbuduj wektor wejsc dla I w.ukrytej (dodanie wej. bias)
   u1=W1'*x1;                       % wyznacz wektor pobudzen I warstwy ukrytej
   y1=1./(1+exp(-beta1*u1));        % wyjscie I warstwy ukrytej; 
                                % funkcja aktywacji I warstwy ukrytej - sigmoidalna unipolarna
   x2 = [-1; y1];                   % zbuduj wektor wejsc dla II w.ukrytej (dodanie wej. bias)
   u2=W2'*x2;                       % wyznacz wektor pobudzen II warstwy ukrytej
   y2=1./(1+exp(-beta2*u2));        % wyjscie II warstwy ukrytej; 
                                % funkcja aktywacji II warstwy ukrytej - sigmoidalna unipolarna
   x3=[-1; y2];                     % zbuduj wektor wejsc dla warstwy wyjsciowej (dodanie wej. bias)
   u3=W3'*x3;                     	% wyznacz wektor pobudzen warstwy wyjsciowej
   y3=1./(1+exp(-beta3*u3));    % wyjscie warstwy wyjsciowej; 
                                % funkcja aktywacji warstwy wyjsciowej -
                                % sigmoidalna unipolarna
   
   ty = baza_ucz_wy(:,L);           % oczekiwany wektor wyjscia dla wektora uczacego
   d3=ty-y3;                        % wektor bledu pomiedzy wyjsciem sieci, a wyjsciem oczekiwanym
   df3=y3.*(1-y3);          % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -w.wyj
   dW3=eta3*x3*(d3.*df3)';          % macierz poprawek dla warstwy wyjsciowej
   W3=W3+dW3;                       % aktualizacja macierzy wag warstwy wyjsciowej

   d2=W3*d3;                        % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df2=x3.*(1-x3);                  % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -II w.ukryta
   dd2=(d2.*df2)';                  % wektor wspolczynnikow bledu 
   dd2=dd2(2:end);                  % skrocenie wektora wspolczynnikow bledu
   dW2=eta2*x2*dd2;                 % macierz poprawek dla II w.ukrytej; wsteczna propagacja bledu
   W2=W2+dW2;
   
   d1=W2*d2(2:end);                 % obliczanie wspolczynnika bledu; wektor sum (blad warstwy wyzszej)*waga
   df1=x2.*(1-x2);                  % wektor pochodnych funkcji aktywacji w punkcie pobudzenia -I w.ukryta
   dd1=(d1.*df1)';                  % wektor wspolczynnikow bledu
   dd1=dd1(2:end);                  % skrocenie wektora wspolczynnikow bledu 
                                % nie mozna poprawiac wektora wag dla
                                % wejscia bias, bo ma ono stala wartosc
   dW1=eta1*x1*dd1;                 % macierz poprawek dla warstwy wyjsciowej; wsteczna propagacja bledu
   W1=W1+dW1;                       % aktualizacja macierzy wag warstwy ukrytej
   
   % ew. dodanie warunku na wybranie najlepszej powstalej instancji dla N
   % iteracji
end
end

%% prezentacja najlepszego wypracowanego rozwiazania
Y_matrix=zeros(size(baza_ucz_wy,1),size(baza_test_we,2));
for przykl = 1 : size(baza_test_we,2) % dla wszystkich przykladow z bazy ucz.
    x1= [-1; baza_test_we(:,przykl)];   % wejscie I  w.ukrytej
    u1=W1'*x1;                          % pobudzenie I w.ukrytej
    y1=1./(1+exp(-beta1*u1));           % wyjscie I w.ukrytej    
    x2=[-1; y1];                        % wejscie II w.ukrytej
    u2=W2'*x2;                          % pobudzenie II w.ukrytej
    y2=1./(1+exp(-beta2*u2));           % wyjscie II w.ukrytej
    x3=[-1; y2];                        % wejscie w.wyjsciowej
    u3=W3'*x3;                          % pobudzenie w.wyjsciowej
    y3=1./(1+exp(-beta3*u3));       % wyjscie w.wyjsciowej
    Y_matrix(:,przykl)=y3;              % zapisz wynik dla tego przykladu w macierzy
end
Y_matrix % wyswietl odpowiedz sieci
MSE=immse(Y_matrix,baza_test_wy) % MSE wzgledem oczekiwanego wyjscia zest. testowego