#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

#include <string>

#include "Definitions.hpp"
#include "PublicKey.hpp"
#include "PrivateKey.hpp"

namespace fideslib {
    enum SerType {
        BINARY = 0,
        JSON   = 1
    };
}

namespace fideslib::Serial {

    // Label on the .dev sidecar's ReducedNoise line, shared by the writer and reader so the two
    // cannot drift apart.
    inline constexpr const char* kReducedNoiseLabel = "ReducedNoise:";

    bool SerializeToFile(const std::string& filename, const fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype);
    bool SerializeToFile(const std::string& filename, const fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
    bool SerializeToFile(const std::string& filename, const fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype);

    bool DeserializeFromFile(const std::string& filename, fideslib::CryptoContext<fideslib::DCRTPoly>& obj, const SerType& sertype);
    bool DeserializeFromFile(const std::string& filename, fideslib::PublicKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
    bool DeserializeFromFile(const std::string& filename, fideslib::PrivateKey<fideslib::DCRTPoly>& obj, const SerType& sertype);
}

#endif // SERIALIZE_HPP