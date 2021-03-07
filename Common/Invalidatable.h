#ifndef RAY_RACING_CLUB_INVALIDATABLE_H
#define RAY_RACING_CLUB_INVALIDATABLE_H

class Invalidatable {
public:
    Invalidatable();

    void setValid(bool flag);
    bool isValid() const;

private:
    bool mIsValid;
};

#endif // RAY_RACING_CLUB_INVALIDATABLE_H
