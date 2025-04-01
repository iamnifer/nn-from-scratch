#include "except.h"
#include <exception>

namespace except {
    void react() {
        try {
            throw;
        } catch(std::exception& e) {
            // known exc
        } catch(...) {
            // unknown exc
        }
    }
}
