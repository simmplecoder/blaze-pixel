#include <blaze/Blaze.h>
#include <boost/gil/typedefs.hpp>

namespace flash
{

template <typename T>
using image_matrix = blaze::CustomMatrix<T, blaze::unaligned, blaze::unpadded>;

/** \brief Used to bypass scoped_channel_type of GIL
    Some channel types in GIL, like `gil::float32_t` and `gil::float64_t` have a different type
    compared to it's underlying float type (because they constrain the range to be [0...1]),
    this struct will strip that and provide typedef equal to the underlying channel type.
*/
template <typename ChannelType>
struct true_channel_type
{
    using type = ChannelType;
};

/// \cond specializations
template <>
struct true_channel_type<boost::gil::float32_t>
{
    using type = float;
};

template <>
struct true_channel_type<boost::gil::float64_t>
{
    using type = double;
};
/// \endcond

template <typename ChannelType>
using true_channel_type_t = typename true_channel_type<ChannelType>::type;

namespace detail
{
template <typename PixelType, std::size_t... indices>
auto pixel_to_vector_impl(const PixelType& pixel, std::integer_sequence<std::size_t, indices...>)
{
    using channel_t = typename boost::gil::channel_type<PixelType>::type;
    return blaze::StaticVector<true_channel_type_t<channel_t>, sizeof...(indices)>{
        pixel[indices]...};
}
} // namespace detail

/** \brief Converts a pixel into `StaticVector`
    Useful when working with multi-channel images
    \tparam PixelType The source pixel type
    \arg pixel The source pixel to convert into `StaticVector`
    \return a `StaticVector` where elements correspond to channel values, in the same order
*/
template <typename PixelType>
auto pixel_to_vector(const PixelType& pixel)
{
    return detail::pixel_to_vector_impl(
        pixel, std::make_index_sequence<boost::gil::num_channels<PixelType>{}>{});
}

/** \brief Converts vector into pixel
    Useful when working with multi-channel images.
    \tparam PixelType The pixel type to convert to
    \tparam VT The concrete vector type
    \arg vector The source vector to convert into `PixelType`
    \tparam TransposeFlag Transpose flag for the vector
    \return A pixel where each channel corresponds to entry in the vector, in the same order
*/
template <typename PixelType, typename VT, bool TransposeFlag>
auto vector_to_pixel(const blaze::DenseVector<VT, TransposeFlag>& vector)
{
    auto num_channels = boost::gil::num_channels<PixelType>{};
    PixelType pixel{};
    for (std::size_t i = 0; i < num_channels; ++i)
    {
        pixel[i] = (~vector)[i];
    }

    return pixel;
}

} // namespace flash