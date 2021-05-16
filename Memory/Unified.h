//
// Created by awesyr on 26.02.2021.
//

#ifndef RAY_RACING_CLUB_UNIFIED_H
#define RAY_RACING_CLUB_UNIFIED_H

class Unified {
public:
    void* operator new(size_t len);
    void operator delete(void *ptr);
};


#endif //RAY_RACING_CLUB_UNIFIED_H
