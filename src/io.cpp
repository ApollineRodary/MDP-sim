#include <iostream>
#include "io.hpp"

#define LENGTH 50

using namespace std;

void show_loading_bar(string text, int a, int max) {
    if (a*LENGTH % max > 0)
        return;

    cout << text << " [";
    for (int i=0; i<LENGTH; i++)
        cout << " ";
    cout << "]\r" << text << " [";
    for (int i=0; i < a*LENGTH/max; i++)
        cout << "=";
    cout << "\r";
    cout.flush();
}