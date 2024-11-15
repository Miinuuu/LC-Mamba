import numpy as np
import torch
# Hilbert Curve 설정
from typing import Iterable, List, Union
import multiprocessing
from multiprocessing import Pool
import numpy as np



def _binary_repr(num: int, width: int) -> str:
    """Return a binary string representation of `num` zero padded to `width`
    bits."""
    return format(num, 'b').zfill(width)


class HilbertCurve:

    def __init__(
        self,
        p: Union[int, float],
        n: Union[int, float],
        n_procs: int=0,
    ) -> None:

        """Initialize a hilbert curve with,

        Args:
            p (int or float): iterations to use in constructing the hilbert curve.
                if float, must satisfy p % 1 = 0
            n (int or float): number of dimensions.
                if float must satisfy n % 1 = 0
            n_procs (int): number of processes to use
                0 = dont use multiprocessing
               -1 = use all available threads
                any other positive integer = number of processes to use

        """
        if (p % 1) != 0:
            raise TypeError("p is not an integer and can not be converted")
        if (n % 1) != 0:
            raise TypeError("n is not an integer and can not be converted")
        if (n_procs % 1) != 0:
            raise TypeError("n_procs is not an integer and can not be converted")

        self.p = int(p)
        self.n = int(n)

        if self.p <= 0:
            raise ValueError('p must be > 0 (got p={} as input)'.format(p))
        if self.n <= 0:
            raise ValueError('n must be > 0 (got n={} as input)'.format(n))

        # minimum and maximum distance along curve
        self.min_h = 0
        self.max_h = 2**(self.p * self.n) - 1

        # minimum and maximum coordinate value in any dimension
        self.min_x = 0
        self.max_x = 2**self.p - 1

        # set n_procs
        n_procs = int(n_procs)
        if n_procs == -1:
            self.n_procs = multiprocessing.cpu_count()
        elif n_procs == 0:
            self.n_procs = 0
        elif n_procs > 0:
            self.n_procs = n_procs
        else:
            raise ValueError(
                'n_procs must be >= -1 (got n_procs={} as input)'.format(n_procs))


    def _hilbert_integer_to_transpose(self, h: int) -> List[int]:
        """Store a hilbert integer (`h`) as its transpose (`x`).

        Args:
            h (int): integer distance along hilbert curve

        Returns:
            x (list): transpose of h
                (n components with values between 0 and 2**p-1)
        """
        h_bit_str = _binary_repr(h, self.p*self.n)
        x = [int(h_bit_str[i::self.n], 2) for i in range(self.n)]
        return x


    def _transpose_to_hilbert_integer(self, x: Iterable[int]) -> int:
        """Restore a hilbert integer (`h`) from its transpose (`x`).

        Args:
            x (list): transpose of h
                (n components with values between 0 and 2**p-1)

        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [_binary_repr(x[i], self.p) for i in range(self.n)]
        h = int(''.join([y[i] for i in range(self.p) for y in x_bit_str]), 2)
        return h


    def point_from_distance(self, distance: int) -> Iterable[int]:
        """Return a point in n-dimensional space given a distance along a hilbert curve.

        Args:
            distance (int): integer distance along hilbert curve

        Returns:
            point (iterable of ints): an n-dimensional vector of lengh n where
            each component value is between 0 and 2**p-1.
        """
        x = self._hilbert_integer_to_transpose(int(distance))
        z = 2 << (self.p-1)

        # Gray decode by H ^ (H/2)
        t = x[self.n-1] >> 1
        for i in range(self.n-1, 0, -1):
            x[i] ^= x[i-1]
        x[0] ^= t

        # Undo excess work
        q = 2
        while q != z:
            p = q - 1
            for i in range(self.n-1, -1, -1):
                if x[i] & q:
                    # invert
                    x[0] ^= p
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & p
                    x[0] ^= t
                    x[i] ^= t
            q <<= 1

        return x


    def points_from_distances(
        self,
        distances: Iterable[int],
        match_type: bool=False,
    ) -> Iterable[Iterable[int]]:
        """Return points in n-dimensional space given distances along a hilbert curve.

        Args:
            distances (iterable of int): iterable of integer distances along hilbert curve
            match_type (bool): if True, make type(points) = type(distances)

        Returns:
            points (iterable of iterable of ints): an iterable of n-dimensional vectors
                where each vector has lengh n and component values between 0 and 2**p-1.
                if match_type=False will be list of lists else type(points) = type(distances)
        """
        for ii, dist in enumerate(distances):
            if (dist % 1) != 0:
                raise TypeError(
                    "all values in distances must be int or floats that are convertible to "
                    "int but found distances[{}]={}".format(ii, dist))
            if dist > self.max_h:
                raise ValueError(
                    "all values in distances must be <= 2**(p*n)-1={} but found "
                    "distances[{}]={} ".format(self.max_h, ii, dist))
            if dist < self.min_h:
                raise ValueError(
                    "all values in distances must be >= {} but found distances[{}]={} "
                    "".format(self.min_h, ii, dist))

        if self.n_procs == 0:
            points = []
            for distance in distances:
                x = self.point_from_distance(distance)
                points.append(x)
        else:
            with Pool(self.n_procs) as p:
                points = p.map(self.point_from_distance, distances)

        if match_type:
            if isinstance(distances, np.ndarray):
                points = np.array(points, dtype=distances.dtype)
            else:
                target_type = type(distances)
                points = target_type([target_type(vec) for vec in points])

        return points


    def distance_from_point(self, point: Iterable[int]) -> int:
        """Return distance along the hilbert curve for a given point.

        Args:
            point (iterable of ints): an n-dimensional vector where each component value
                is between 0 and 2**p-1.

        Returns:
            distance (int): integer distance along hilbert curve
        """
        point = [int(el) for el in point]

        m = 1 << (self.p - 1)

        # Inverse undo excess work
        q = m
        while q > 1:
            p = q - 1
            for i in range(self.n):
                if point[i] & q:
                    point[0] ^= p
                else:
                    t = (point[0] ^ point[i]) & p
                    point[0] ^= t
                    point[i] ^= t
            q >>= 1

        # Gray encode
        for i in range(1, self.n):
            point[i] ^= point[i-1]
        t = 0
        q = m
        while q > 1:
            if point[self.n-1] & q:
                t ^= q - 1
            q >>= 1
        for i in range(self.n):
            point[i] ^= t

        distance = self._transpose_to_hilbert_integer(point)
        return distance


    def distances_from_points(
        self,
        points: Iterable[Iterable[int]],
        match_type: bool=False,
    ) -> Iterable[int]:
        """Return distances along the hilbert curve for a given set of points.

        Args:
            points (iterable of iterable of ints): an iterable of n-dimensional vectors
                where each vector has lengh n and component values between 0 and 2**p-1.
            match_type (bool): if True, make type(distances) = type(points)

        Returns:
            distances (iterable of int): iterable of integer distances along hilbert curve
              the return type will match the type used for points.
        """
        for ii, point in enumerate(points):

            if len(point) != self.n:
                raise ValueError(
                    "all vectors in points must have length n={} "
                    "but found points[{}]={}".format(self.n, ii, point))

            if any(elx > self.max_x for elx in point):
                raise ValueError(
                    "all coordinate values in all vectors in points must be <= 2**p-1={} "
                    "but found points[{}]={}".format(self.max_x, ii, point))

            if any(elx < self.min_x for elx in point):
                raise ValueError(
                    "all coordinate values in all vectors in points must be > {} "
                    "but found points[{}]={}".format(self.min_x, ii, point))

            if any((elx % 1) != 0 for elx in point):
                raise TypeError(
                    "all coordinate values in all vectors in points must be int or floats "
                    "that are convertible to int but found points[{}]={}".format(ii, point))

        if self.n_procs == 0:
            distances = []
            for point in points:
                distance = self.distance_from_point(point)
                distances.append(distance)
        else:
            with Pool(self.n_procs) as p:
                distances = p.map(self.distance_from_point, points)

        if match_type:
            if isinstance(points, np.ndarray):
                distances = np.array(distances, dtype=points.dtype)
            else:
                target_type = type(points)
                distances = target_type(distances)

        return distances


    def __str__(self):
        return f"HilbertCruve(p={self.p}, n={self.n}, n_procs={self.n_procs})"


    def __repr__(self):
        return self.__str__()

def apply_hilbert_curve_2d(tensor,sorted_indices):
    """
    입력:
        tensor: (B, C, H, W) 형태의 텐서
    출력:
        hilbert_tensor: 힐베르트 곡선 순서로 재배열된 텐서, shape은 (B, C, N)
        inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
    """
    B, C, H, W = tensor.shape
    # L = H * W  # 픽셀 수
    # # 힐베르트 곡선의 차원과 단계 설정
    # p = int(np.ceil(np.log2(max(H, W))))  # 힐베르트 곡선의 단계 (order)
    # n = 2  # 2차원

    # # 힐베르트 곡선 객체 생성
    # hilbert_curve = HilbertCurve(p, n)

    # # 힐베르트 곡선의 전체 좌표 계산
    # coords = []
    # for y in range(H):
    #     for x in range(W):
    #         coords.append((x, y))

    # # 각 좌표에 대한 힐베르트 인덱스 계산
    # hilbert_indices = []
    # for coord in coords:
    #     x, y = coord
    #     # 힐베르트 곡선의 크기에 맞게 좌표 조정
    #     hilbert_index = hilbert_curve.distance_from_point([x, y])
    #     hilbert_indices.append(hilbert_index)

    # # 힐베르트 인덱스에 따라 정렬
    # hilbert_indices = np.array(hilbert_indices)
    # sorted_indices = np.argsort(hilbert_indices)
    # # 역순서 인덱스 계산
    # inverse_indices = np.argsort(sorted_indices)

    # 입력 텐서를 힐베르트 순서로 재배열
    tensor_flat = tensor.view(B,  C, -1) # (B,K, C, H*W)
    hilbert_tensor = tensor_flat[: , : , sorted_indices] # (B,K, C,L)

    return hilbert_tensor.contiguous() 
    #, inverse_indices

# def apply_hilbert_curve(tensor):
#     """
#     입력:
#         tensor: (B, C, H, W) 형태의 텐서
#     출력:
#         hilbert_tensor: 힐베르트 곡선 순서로 재배열된 텐서, shape은 (B, C, N)
#         inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
#     """
#     B, K,C, H, W = tensor.shape
#     L = H * W  # 픽셀 수
#     # 힐베르트 곡선의 차원과 단계 설정
#     p = int(np.ceil(np.log2(max(H, W))))  # 힐베르트 곡선의 단계 (order)
#     n = 2  # 2차원

#     # 힐베르트 곡선 객체 생성
#     hilbert_curve = HilbertCurve(p, n)

#     # 힐베르트 곡선의 전체 좌표 계산
#     coords = []
#     for y in range(H):
#         for x in range(W):
#             coords.append((x, y))

#     # 각 좌표에 대한 힐베르트 인덱스 계산
#     hilbert_indices = []
#     for coord in coords:
#         x, y = coord
#         # 힐베르트 곡선의 크기에 맞게 좌표 조정
#         hilbert_index = hilbert_curve.distance_from_point([x, y])
#         hilbert_indices.append(hilbert_index)

#     # 힐베르트 인덱스에 따라 정렬
#     hilbert_indices = np.array(hilbert_indices)
#     sorted_indices = np.argsort(hilbert_indices)
#     # 역순서 인덱스 계산
#     inverse_indices = np.argsort(sorted_indices)

#     # 입력 텐서를 힐베르트 순서로 재배열
#     tensor_flat = tensor.view(B, K, C, -1)  # (B,K, C, H*W)
#     hilbert_tensor = tensor_flat[:, : , : , sorted_indices]  # (B,K, C,L)

#     return hilbert_tensor, inverse_indices

# def reverse_hilbert_curve(hilbert_tensor, inverse_indices, H, W):
#     """
#     입력:
#         hilbert_tensor: 힐베르트 곡선 순서로 정렬된 텐서, shape은 (B, C, N)
#         inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
#         H, W: 원래 이미지의 높이와 너비
#     출력:
#         tensor: 원래의 (B, C, H, W) 형태의 텐서
#     """
#     B, K, C, N = hilbert_tensor.shape
#     # 원래 순서로 재배열
#     tensor_flat = hilbert_tensor[:, :, :,inverse_indices]  # (B, K,C, N)
#     # 원래의 이미지 형태로 변환
#     tensor = tensor_flat.view(B,K, C, H, W)
#     return tensor
def reverse_hilbert_curve_2d(hilbert_tensor, inverse_indices, H, W):
    """
    입력:
        hilbert_tensor: 힐베르트 곡선 순서로 정렬된 텐서, shape은 (B, C, N)
        inverse_indices: 원래 순서로 복원하기 위한 인덱스 배열
        H, W: 원래 이미지의 높이와 너비
    출력:
        tensor: 원래의 (B, C, H, W) 형태의 텐서
    """
    B, C, N = hilbert_tensor.shape
    # 원래 순서로 재배열
    tensor_flat = hilbert_tensor[:, :,inverse_indices] # (B, K,C, N)
    # 원래의 이미지 형태로 변환
    tensor = tensor_flat.view(B, C, H, W)
    return tensor.contiguous()

# 예시 사용법
if __name__ == "__main__":
    B, C, H, W = 1, 3, 8, 8  # 예제 크기


    p = int(np.log2(8))  # 힐베르트 곡선의 단계 (order)
    n = 2  # 2차원

    H,W=8,8

    # 힐베르트 곡선 객체 생성
    hilbert_curve = HilbertCurve(p, n)

    # 힐베르트 곡선의 전체 좌표 계산
    coords = []
    for y in range(H):
        for x in range(W):
            coords.append((x, y))

    # 각 좌표에 대한 힐베르트 인덱스 계산
    hilbert_indices = []
    for coord in coords:
        x, y = coord
        # 힐베르트 곡선의 크기에 맞게 좌표 조정
        hilbert_index = hilbert_curve.distance_from_point([x, y])
        hilbert_indices.append(hilbert_index)

    # 힐베르트 인덱스에 따라 정렬
    hilbert_indices = np.array(hilbert_indices)
    sorted_indices = np.argsort(hilbert_indices)
    # 역순서 인덱스 계산
    inverse_indices = np.argsort( sorted_indices)





    tensor = torch.randn(B, C, H, W)
    tensor_wh = torch.transpose(tensor,-1,-2)

    # 힐베르트 곡선 적용
    hilbert_tensor  = apply_hilbert_curve_2d(tensor,sorted_indices)
    print("힐베르트 곡선 적용 후 텐서 크기:", hilbert_tensor.shape)

    # 원래의 이미지로 복원
    restored_tensor = reverse_hilbert_curve_2d(hilbert_tensor, inverse_indices, H, W)
    print("복원된 텐서 크기:", restored_tensor.shape)

    # 복원 결과 확인
    if torch.allclose(tensor, restored_tensor):
        print("복원이 정확하게 이루어졌습니다.")
    else:
        print("복원에 오류가 있습니다.")