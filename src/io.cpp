#include <iostream>
#include "io.hpp"

using namespace std;

void show_loading_bar(string text, int a, int max) {
    if (a*100 % max > 0)
        return;

    cout << text << " [";
    for (int i=0; i<100; i++)
        cout << " ";
    cout << "]\r" << text << " [";
    for (int i=0; i < a*100/max; i++)
        cout << "=";
    cout << "\r";
    cout.flush();
}