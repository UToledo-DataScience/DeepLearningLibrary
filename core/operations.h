#ifndef OPERATIONS
#define OPERATIONS
#include <iostream>

template <typename OpDType>
class Operation {
    protected:
        std::string name;
        std::string type;

        Operation<OpDType>* parent1;
        Operation<OpDType>* parent2;

        Placeholder<OpDType>* placeholder;

    public:
        Operation() {
            parent1 = nullptr;
            parent2 = nullptr;
            placeholder = nullptr;
        }

        Operation(Operation<OpDType>* p1, Operation<OpDType>* p2);
        Operation(Placeholder<OpDType>* pl);

        // assuming that each operation can have a max of two parents
        virtual OpDType derive() = 0;
        virtual OpDType operate() = 0;

        std::string getType();
};

// NOTE: this is only rigged for tensors with one element
template <typename OpDType>
class Multiplication : public Operation<OpDType> {
    public:
        Multiplication(Operation<OpDType>* p1, Operation<OpDType>* p2) {
            this->parent1 = p1;
            this->parent2 = p2;
            this->type = "multiplication";
        }

        OpDType derive() {
            OpDType a = this->parent1->operate() * this->parent2->derive();
            OpDType b = this->parent2->operate() * this->parent1->derive();

            return a + b;
        }

        OpDType operate() {
            OpDType p1, p2;

            if (this->parent1 != nullptr)
                p1 = this->parent1->operate();

            if (this->parent2 != nullptr)
                p2 = this->parent2->operate();

            return p1 * p2;
        }
};

template <typename OpDType>
class Constant : public Operation<OpDType> {
    public:
        Constant(Placeholder<OpDType>* pl) {
            this->placeholder = pl;
            this->type = "constant";
        }

        OpDType derive() {
            return 0;
        }

        OpDType operate() {
            return this->placeholder->getIndex(0);
        }
};
#endif
