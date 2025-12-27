#include <iostream>
#include <sstream>

int main(int argc, char* argv[]) {
    if(argc <= 1) {
        std::cerr << "Usage: .\\Main.exe <lab_number>";
        return -1;
    }

    // assert(argc >= 2)
    std::istringstream strNum(std::string{argv[1]});
    int labNum = -1;
    strNum >> labNum;

    switch (labNum)
    {
    default:
        std::cerr << "Usage: <lab_number> not exist";
        return -1;
    }
    
    return 0;
}