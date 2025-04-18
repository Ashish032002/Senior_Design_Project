"""
This type stub file was generated by pyright.
"""

import dataclasses
import google.ai.generativelanguage as glm
from typing import Any, AsyncIterable, Iterable, Optional, Union
from google.generativeai import protos, string_utils

"""
This type stub file was generated by pyright.
"""
__all__ = ["Permission", "Permissions"]
GranteeType = protos.Permission.GranteeType
Role = protos.Permission.Role
GranteeTypeOptions = Union[str, int, GranteeType]
RoleOptions = Union[str, int, Role]
_GRANTEE_TYPE: dict[GranteeTypeOptions, GranteeType] = ...
_ROLE: dict[RoleOptions, Role] = ...
_VALID_PERMISSION_ID = ...
INVALID_PERMISSION_ID_MSG = ...
def to_grantee_type(x: GranteeTypeOptions) -> GranteeType:
    ...

def to_role(x: RoleOptions) -> Role:
    ...

def valid_id(name: str) -> bool:
    ...

@string_utils.prettyprint
@dataclasses.dataclass(init=False)
class Permission:
    """
    A permission to access a resource.
    """
    name: str
    role: Role
    grantee_type: Optional[GranteeType]
    email_address: Optional[str] = ...
    def __init__(self, name: str, role: RoleOptions, grantee_type: Optional[GranteeTypeOptions] = ..., email_address: Optional[str] = ...) -> None:
        ...
    
    def delete(self, client: glm.PermissionServiceClient | None = ...) -> None:
        """
        Delete permission (self).
        """
        ...
    
    async def delete_async(self, client: glm.PermissionServiceAsyncClient | None = ...) -> None:
        """
        This is the async version of `Permission.delete`.
        """
        ...
    
    def update(self, updates: dict[str, Any], client: glm.PermissionServiceClient | None = ...) -> Permission:
        """
        Update a list of fields for a specified permission.

        Args:
            updates: The list of fields to update.
                     Currently only `role` is supported as an update path.

        Returns:
            `Permission` object with specified updates.
        """
        ...
    
    async def update_async(self, updates: dict[str, Any], client: glm.PermissionServiceAsyncClient | None = ...) -> Permission:
        """
        This is the async version of `Permission.update`.
        """
        ...
    
    def to_dict(self) -> dict[str, Any]:
        ...
    
    @classmethod
    def get(cls, name: str, client: glm.PermissionServiceClient | None = ...) -> Permission:
        """
        Get information about a specific permission.

        Args:
            name: The name of the permission to get.

        Returns:
            Requested permission as an instance of `Permission`.
        """
        ...
    
    @classmethod
    async def get_async(cls, name: str, client: glm.PermissionServiceAsyncClient | None = ...) -> Permission:
        """
        This is the async version of `Permission.get`.
        """
        ...
    


class Permissions:
    def __init__(self, parent) -> None:
        ...
    
    @property
    def parent(self):
        ...
    
    def create(self, role: RoleOptions, grantee_type: Optional[GranteeTypeOptions] = ..., email_address: Optional[str] = ..., client: glm.PermissionServiceClient | None = ...) -> Permission:
        """
        Create a new permission on a resource (self).

        Args:
            parent: The resource name of the parent resource in which the permission will be listed.
            role: role that will be granted by the permission.
            grantee_type: The type of the grantee for the permission.
            email_address: The email address of the grantee.

        Returns:
            `Permission` object with specified parent, role, grantee type, and email address.

        Raises:
            ValueError: When email_address is specified and grantee_type is set to EVERYONE.
            ValueError: When email_address is not specified and grantee_type is not set to EVERYONE.
        """
        ...
    
    async def create_async(self, role: RoleOptions, grantee_type: Optional[GranteeTypeOptions] = ..., email_address: Optional[str] = ..., client: glm.PermissionServiceAsyncClient | None = ...) -> Permission:
        """
        This is the async version of `PermissionAdapter.create_permission`.
        """
        ...
    
    def list(self, page_size: Optional[int] = ..., client: glm.PermissionServiceClient | None = ...) -> Iterable[Permission]:
        """
        List `Permission`s enforced on a resource (self).

        Args:
            parent: The resource name of the parent resource in which the permission will be listed.
            page_size: The maximum number of permissions to return (per page). The service may return fewer permissions.

        Returns:
            Paginated list of `Permission` objects.
        """
        ...
    
    def __iter__(self):
        ...
    
    async def list_async(self, page_size: Optional[int] = ..., client: glm.PermissionServiceAsyncClient | None = ...) -> AsyncIterable[Permission]:
        """
        This is the async version of `PermissionAdapter.list_permissions`.
        """
        ...
    
    async def __aiter__(self):
        ...
    
    @classmethod
    def get(cls, name: str) -> Permission:
        """
        Get information about a specific permission.

        Args:
            name: The name of the permission to get.

        Returns:
            Requested permission as an instance of `Permission`.
        """
        ...
    
    @classmethod
    async def get_async(cls, name: str) -> Permission:
        """
        Get information about a specific permission.

        Args:
            name: The name of the permission to get.

        Returns:
            Requested permission as an instance of `Permission`.
        """
        ...
    
    def transfer_ownership(self, email_address: str, client: glm.PermissionServiceClient | None = ...) -> None:
        """
        Transfer ownership of a resource (self) to a new owner.

        Args:
            name: Name of the resource to transfer ownership.
            email_address: Email address of the new owner.
        """
        ...
    
    async def transfer_ownership_async(self, email_address: str, client: glm.PermissionServiceAsyncClient | None = ...) -> None:
        """This is the async version of `PermissionAdapter.transfer_ownership`."""
        ...
    


