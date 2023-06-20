#include <boost/python.hpp>
#include <vector>
#include "../src/mdp.hpp" 

using namespace boost::python;
using namespace std;

class Bonjour {
    public:
    Bonjour(int a) {
        cout << a << endl;
    }

    int prompt() {
        cout << "Bonjour !" << endl;
        return 3;
    }
};

BOOST_PYTHON_MODULE(pymdp) {
    class_<Bonjour>("Bonjour", init<int>())
    .def("prompt", &Bonjour::prompt);
}
