//
// Created by awesyr on 26.02.2021.
//

#ifndef RAY_RACING_CLUB_MANAGED_H
#define RAY_RACING_CLUB_MANAGED_H


class Managed {
public:
    void* operator new(size_t len);
    void operator delete(void *ptr);
};


#endif //RAY_RACING_CLUB_MANAGED_H
