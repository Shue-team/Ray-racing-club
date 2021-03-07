#include "Invalidatable.h"

Invalidatable::Invalidatable() : mIsValid(true) {}

void Invalidatable::setValid(bool flag) {
    mIsValid = flag;
}

bool Invalidatable::isValid() const
{
    return mIsValid;
}
