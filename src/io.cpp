#include <iostream>
#include "io.hpp"

#define LENGTH 50

using namespace std;

void show_loading_bar(const char text[20], int a, int max) {
    if (a*LENGTH % max > 0)
        return;

    int n = a*LENGTH/max;

    cout << "\r                    [";
    for (int i=0; i<n; i++)
        cout << "=";
    for (int i=n; i<LENGTH; i++)
        cout << " ";
    cout << "]\r";

    cout << text;
    cout.flush();

    if (a==max)
        cout << endl;
}