#include <iostream>
#include <sstream>
#include <book_cpp.hpp>

// collection of chapter3 codes
int ch_3(void);
// collection of chapter4 codes
int ch_4(void);

int main(int argc, char* argv[]) {
    if(argc <= 1) {
        std::cerr << "Usage: .\\Main.exe <lab_number>";
        return -1;
    }

    // assert(argc >= 2)
    std::istringstream strNum(std::string{argv[1]});
    int labNum = -1;
    strNum >> labNum;

    int stat = 0;

    switch (labNum)
    {
    case 3:
        stat = ch_3();
        break;
    case 4:
        stat = ch_4();
        break;
    default:
        std::cerr << "Usage: <lab_number> not exist";
        return -1;
    }

    return stat;
}