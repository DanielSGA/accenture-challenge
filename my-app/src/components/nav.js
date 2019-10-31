import React from 'react';
import { Navbar, Nav as NavBootstrap, NavItem, NavDropdown, MenuItem } from 'react-bootstrap';
export const Nav = (props)=>{
    return(
        <Navbar className="justify-content-md-center" bg="light" expand="lg">
        <Navbar.Brand href="#home" className="principal">{props.text1}</Navbar.Brand>
        </Navbar>
    )
}