from .matching_zone import matching_zone
from common import zipped_mzone


class matching_zone_repository():
    
    @classmethod
    def init(cls):
        cls.mzone_repo = [matching_zone(m_id,hex_ids) for m_id,hex_ids in zipped_mzone.get_mzone_info()]
    
    @classmethod
    def get_all(cls):
        return cls.mzone_repo

